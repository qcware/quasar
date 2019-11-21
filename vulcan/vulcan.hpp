#pragma once

#include "vulcan_gpu.hpp"
#include "vulcan_types.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>

namespace vulcan {    

namespace util {

template <typename T>
std::vector<T> identity2()
{
    std::vector<T> I(4, T(0.0));
    I[0] = T(1.0);
    I[3] = T(1.0);
    return I;
}

template <typename T>
std::vector<T> identity4()
{
    std::vector<T> I(16, T(0.0));
    I[0]  = T(1.0);
    I[5]  = T(1.0);
    I[10] = T(1.0);
    I[15] = T(1.0);
    return I;
}

template <typename T>
std::vector<T> multiply2(
    const std::vector<T>& U1,
    const std::vector<T>& U2)
{
    if (U1.size() != 4) throw std::runtime_error("U1 size != 4");    
    if (U2.size() != 4) throw std::runtime_error("U2 size != 4");    
    std::vector<T> U(4);
    U[0] = U1[0] * U2[0] + U1[1] * U2[2];
    U[1] = U1[0] * U2[1] + U1[1] * U2[3]; 
    U[2] = U1[2] * U2[0] + U1[3] * U2[2];
    U[3] = U1[2] * U2[1] + U1[3] * U2[3]; 
    return U;
}

template <typename T>
std::vector<T> multiply4(
    const std::vector<T>& U1,
    const std::vector<T>& U2)
{
    if (U1.size() != 16) throw std::runtime_error("U1 size != 16");    
    if (U2.size() != 16) throw std::runtime_error("U2 size != 16");    
    std::vector<T> U(16, T(0.0));
    for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
        U[i*4 + j] += U1[i*4 + k] * U2[k*4 + j];
    }}}
    return U;
}

template <typename T>
std::vector<T> kron2(
    const std::vector<T>& U1,
    const std::vector<T>& U2)
{
    if (U1.size() != 4) throw std::runtime_error("U1 size != 4");    
    if (U2.size() != 4) throw std::runtime_error("U2 size != 4");    
    std::vector<T> U(16); 
    U[ 0] = U1[0] * U2[0];
    U[ 1] = U1[0] * U2[1];
    U[ 2] = U1[1] * U2[0];
    U[ 3] = U1[1] * U2[1];
    U[ 4] = U1[0] * U2[2];
    U[ 5] = U1[0] * U2[3];
    U[ 6] = U1[1] * U2[2];
    U[ 7] = U1[1] * U2[3];
    U[ 8] = U1[2] * U2[0];
    U[ 9] = U1[2] * U2[1];
    U[10] = U1[3] * U2[0];
    U[11] = U1[3] * U2[1];
    U[12] = U1[2] * U2[2];
    U[13] = U1[2] * U2[3];
    U[14] = U1[3] * U2[2];
    U[15] = U1[3] * U2[3];
    return U;
}

template <typename T>
std::vector<T> kron2I1(
    const std::vector<T>& U2)
{
    return kron2<T>(identity2<T>(), U2);
}

template <typename T>
std::vector<T> kron2I2(
    const std::vector<T>& U1)
{
    return kron2<T>(U1, identity2<T>());
}

template <typename T>
std::vector<T> bitflip4(
    const std::vector<T>& U)    
{
    if (U.size() != 16) throw std::runtime_error("U size != 16");    
    std::vector<T> U2(16);
    U2[0*4 + 0] = U[0*4 + 0];
    U2[0*4 + 1] = U[0*4 + 2];
    U2[0*4 + 2] = U[0*4 + 1];
    U2[0*4 + 3] = U[0*4 + 3];
    U2[1*4 + 0] = U[2*4 + 0];
    U2[1*4 + 1] = U[2*4 + 2];
    U2[1*4 + 2] = U[2*4 + 1];
    U2[1*4 + 3] = U[2*4 + 3];
    U2[2*4 + 0] = U[1*4 + 0];
    U2[2*4 + 1] = U[1*4 + 2];
    U2[2*4 + 2] = U[1*4 + 1];
    U2[2*4 + 3] = U[1*4 + 3];
    U2[3*4 + 0] = U[3*4 + 0];
    U2[3*4 + 1] = U[3*4 + 2];
    U2[3*4 + 2] = U[3*4 + 1];
    U2[3*4 + 3] = U[3*4 + 3];
    return U2;
}

} // namespace util

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
typedef std::chrono::high_resolution_clock Clock;
auto t1 = Clock::now();

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

auto t2 = Clock::now();
std::chrono::duration<double> elapsed = t2 - t1;
printf("Time elapsed = %11.3E\n", elapsed.count());

printf("%zu %zu\n", gates1.size(), gates0f.size());

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

/**
 * Apply a Pauli operator to statevector1_d, placing the result in
 * statevector2_d, using statevector3_d as a temporary buffer array:
 *
 *  statevector2_d = Pauli (statevector1_d)
 *
 * NOTE: we use apply_pauli_1_mody with adjusted coefficients to apply the
 * individual 1-body Pauli operators. This yields correct results with scalar
 * types if the Pauli operator is real symmetric (has real coefficients and
 * only even powers of Y).
 *
 * Params:
 *  pauli (Pauli<T>) - Pauli operator to apply
 *  statevector1_d (T*) - device pointer to allocated input statevector (not
 *    modified)
 *  statevector2_d (T*) - device pointer to allocated input statevector
 *    (overwritten)
 *  statevector3_d (T*) - device pointer to allocated input statevector
 *    (overwritten)
 *  Result:
 *    statevector2_d is overwritten with the result
 *    statevector3_d is overwritten with temporary working data
 **/
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
