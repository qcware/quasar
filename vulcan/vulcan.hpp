#pragma once

#include "vulcan_gpu.hpp"
#include "vulcan_types.hpp"
#include "vulcan_util.hpp"
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>

// ==> Project Vulcan: Mid-Level C++11 CPU API <== // 

/**
 * Project Vulcan provides a C++11 API for CUDA-accelerated quantum circuit
 * simulations. This particular part of the API provides a mid-level C++11 API
 * for specifying and operating on quantum circuits (Circuit objects, composed
 * of an array of Gate objects), Pauli operators (Pauli objects), statevectors
 * (2**nqubit-sized arrays of primitive T types), and measurements
 * (nmeasurement-sized arrays of ints).
 *
 * All data fields for this portion of the API are to be provided in host
 * memory (often emphasized with the postfix _h). Memory transfers to/from
 * device memory will be managed by the API routines.
 *
 * Special numerical types are necessary for interoperability between host and
 * device code. These are defined in vulcan_types.hpp. Unless otherwise
 * indicated, the templated functions below are instantiated for the following
 * types T:
 *
 *   vulcan::float32
 *   vulcan::float64
 *   vulcan::complex64
 *   vulcan::complex128
 **/
namespace vulcan {    

// => Vulcan C++ Data Structures <= //

/**
 * Class Gate represents a primitive quantum Gate operation, essentially a
 * named (2**nqubit x 2**nqubit) matrix operator (typically unitary).
 **/ 
template <typename T>
class Gate {

public:
    
/**
 * Verbatim constructor, see fields below
 **/
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

/// Number of qubits in the gate
int nqubit() const { return nqubit_; }
/// Name of the gate, e.g., "Ry"
const std::string& name() const { return name_; }
/// (2**nqubit x 2**nqubit) operator matrix of the gate, in unrolled C order
const std::vector<T>& matrix() const { return matrix_; }

/**
 * The adjoint of the gate, a Gate object with the "matrix" field transposed
 * and conjugated.
 *
 * Returns:
 *  (Gate) - the adjoint gate. Note that only the "matrix" field is updated
 *  (with the adjoint matrix of this gate), the nqubit and name fields are not
 *  changed.
 **/
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

/**
 * Class Circuit represents a quantum circuit of Gate operations, essentially a
 * time-ordered list of Gate objects, qubit indices for these gate objects, and
 * total number of qubits. The Circuit object always has nqubit qubits ordered
 * [0, nqubit).
 **/ 
template <typename T>
class Circuit {

public:

/**
 * Verbatim constructor, see fields below
 **/
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

/// Total number of qubits in this circuit
int nqubit() const { return nqubit_; }
/// Gate objects in this circuit, in time-based order
const std::vector<Gate<T>>& gates() const { return gates_; }
/// Qubit indices of Gate objects in this circuit, in time-based order
const std::vector<std::vector<int>>& qubits() const { return qubits_; }

/**
 * The adjoint of this Circuit.
 *
 * That is, a circuit with gates in time-reversed order, with each gate
 * adjointed.
 *
 * Return:
 *  (Circuit) - the adjoint circuit
 **/
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

/**
 * A version of this Circuit with the endian order of the qubits reversed. 
 *
 * This method is used to transform a Circuit from Mike and Ike order |ABC...>
 * (cirq, quasar) to lexical qubit order |...CBA> (qiskit, vulcan). The
 * transformation is involutary, and can be called twice to obtain the original
 * Circuit.
 *
 * Note that only the "qubits_" field of the new Circuit is updated with new 
 * qubit indices according to the formula:
 *
 *      qubit <- nqubit - 1 - qubit
 *
 * The Gate objects in the Circuit are not modified. 
 *
 * Returns:
 *  (Circuit) - bit-reversed circuit, with qubit indices replaced by
 *      bit-reversed counterparts.
 **/
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

/**
 * An equivalent Circuit with runs of 1- and 2-body gates jammed together into
 * fewer gate objects.
 *
 * Steps:
 *  - runs of 1-body gates are jammed together
 *  - 1-body gates are jammed into neighboring 2-body gates
 *  - runs of 2-body gates are jammed together
 *  - isolated pairs of 1-body gates are wedged into single 2-body gates
 *
 * Do to the nature of the routine, the resultant circuit has either zero or
 * one 1-body gates remaining (depending on if the number of isolated 1-body
 * gates is even or odd, respectively). 
 *
 * Returns:
 *  (Circuit) - compressed circuit. All compressed gates are named "U1" or
 *   "U2". Other gates are unmodified.
 **/
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

/**
 * Class Pauli represents a sparse Pauli operator of the form,
 *
 *  \hat P = \sum_{i} w_i \hat S_i
 *
 * Where \hat S_i is a pauli string, w_i is a weight, and i ranges over [0,
 * nstring). Each Pauli string is a product of X/Y/Z 1-body Pauli operators
 * acting on qubit index k,
 *
 *  \hat S_i = \prod D[i]_k[i]
 *
 * Where the characters D (0-X, 1-Y, 2-Z) and qubit indices k are determined by
 * the individual pauli string. For instance, one might encounter, for a
 * 2-qubit system,
 *
 * I + Z[0] + X[0] * Z[1] + 0.2 * Y[0] * Y[1]
 *
 * This would be written as,
 *
 * Pauli(
 *  2,
 *  { {}, {2}, {0, 2}, {1, 1} },
 *  { {}, {0}, {0, 1}, {0, 1} },
 *  { {1.0}, {1.0}, {1.0}, {0.2} })
 *
 * The Pauli object always has nqubit qubits ordered [0, nqubit).
 *
 * Note that one often encounters the case of Pauli operators corresponding to
 * real, symmetric operators. These are specified by Pauli operators with even
 * numbers of 1-body Y operators in each Pauli string and by real weights. The
 * Vulcan library is designed to produce correct results in such cases when
 * using real types T *if* the corresponding statevectors (e.g., from quantum
 * circuit operations) are also real-valued. This is done by substituting Y ->
 * iY in the application of each Pauli gate, and separately accounting for the
 * factors of (-i)^ny in the weight portion of the Pauli.
 **/ 
template <typename T>
class Pauli {

public:

/**
 * Verbatim constructor, see fields below
 **/
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

/// Total number of qubits in this pauli
int nqubit() const { return nqubit_; }
/// Characters of each Pauli string in this pauli, 0 (X), 1 (Y), or 2 (Z)
const std::vector<std::vector<int>>& types() const { return types_; }
/// Qubit indices of each Pauli string in this pauli
const std::vector<std::vector<int>>& qubits() const { return qubits_; }
/// Weights of each Pauli string
const std::vector<T>& values() const { return values_; }

/**
 * A version of this Pauli with the endian order of the qubits reversed. 
 *
 * This method is used to transform a Pauli from Mike and Ike order |ABC...>
 * (cirq, quasar) to lexical qubit order |...CBA> (qiskit, vulcan). The
 * transformation is involutary, and can be called twice to obtain the original
 * Pauli.
 *
 * Note that only the "qubits_" field of the new Pauli is updated with new 
 * qubit indices according to the formula:
 *
 *      qubit <- nqubit - 1 - qubit
 *
 * Returns:
 *  (Pauli) - bit-reversed pauli, with qubit indices replaced by
 *      bit-reversed counterparts.
 **/
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

/**
 * Run a quantum circuit and return the resulting statevector to the host.
 *
 * GPU memory:
 *  1x statevector
 *
 * Host-device transfers:
 *  2x statevector if input statevector_h is provided 
 *  1x statevector if input statevector_h is NULL
 *
 * Params:
 *  circuit (Circuit) - circuit to run
 *  statevector_h (T*) - input statevector or NULL to indicate the |000...>
 *      statevector
 *  result_h (T*) - output register for final statevector
 *  compressed (bool) - apply compression to the circuit (true) or not (false)?
 * Result:
 *  result_h is overwritten with the final statevector 
 **/
template <typename T>
void run_statevector(
    const Circuit<T>& circuit,
    T* statevector_h,
    T* result_h,
    bool compressed)
{
    T* statevector_d = vulcan::gpu::malloc_and_initialize_statevector(circuit.nqubit(), statevector_h);
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector_d);
    vulcan::gpu::copy_statevector_to_host(circuit.nqubit(), statevector_d, result_h);
    vulcan::gpu::free_statevector(statevector_d);
}

/**
 * Apply a Pauli operator to a statevector and return the resulting statevector
 * to the host.
 *
 * GPU memory:
 *  3x statevector
 *
 * Host-device transfers:
 *  2x statevector if input statevector_h is provided 
 *  1x statevector if input statevector_h is NULL
 *
 * Params:
 *  circuit (Circuit) - circuit to run
 *  statevector_h (T*) - input statevector or NULL to indicate the |000...>
 *      statevector
 *  result_h (T*) - output register for final statevector
 *  compressed (bool) - apply compression to the Pauli (true) or not (false)?
 * Result:
 *  result_h is overwritten with the final statevector 
 **/
template <typename T>
void run_pauli_sigma(
    const Pauli<T>& pauli,
    T* statevector_h,
    T* result_h,
    bool compressed)
{
    T* statevector1_d = vulcan::gpu::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::apply_pauli(pauli, statevector1_d, statevector2_d, statevector3_d);
    vulcan::gpu::copy_statevector_to_host(pauli.nqubit(), statevector2_d, result_h);
    vulcan::gpu::free_statevector(statevector1_d);
    vulcan::gpu::free_statevector(statevector2_d);
    vulcan::gpu::free_statevector(statevector3_d);
}

/**
 * Run a quantum circuit and evaluate the expectation value of the resultant
 * statevector over the various Pauli strings of a sparse Pauli operator. The
 * operation is,
 *
 * { S_i } = { <psi | \hat S_i | psi > }
 *
 * Where { \hat S_i } are the Pauli strings of the input pauli, and psi is the
 * statevector obtained by running the input circuit on the input
 * statevector_h. { S_i } are the scalar Pauli expectation values of each Pauli
 * string.
 *
 * GPU memory:
 *  2x statevector
 *
 * Host-device transfers:
 *  1x statevector if input statevector_h is provided 
 *  Negligible if input statevector_h is NULL
 *
 * Params:
 *  circuit (Circuit) - circuit to run
 *  pauli (Pauli) - pauli operator indicating strings to evaluate expectation
 *      values for (weight values are ignored).
 *  statevector_h (T*) - input statevector or NULL to indicate the |000...>
 *      statevector
 *  compressed (bool) - apply compression to the circuit (true) or not (false)?
 * Returns: 
 *  (Pauli) - pauli object with equivalent strings as input pauli, but with
 *      values fields overwritten with <psi|\hat S_i|psi> expectation values
 *      for each Pauli string \hat S_i. |psi> is the statevector obtained by
 *      running the circuit on statevector_h.
 **/
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
    
    T* statevector1_d = vulcan::gpu::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector1_d);
    Pauli<T> expectation = vulcan::util::pauli_expectation(pauli, statevector1_d, statevector2_d);
    vulcan::gpu::free_statevector(statevector1_d);
    vulcan::gpu::free_statevector(statevector2_d);
    return expectation; 
}

/**
 * Run a quantum circuit and evaluate the expectation value of the resultant
 * statevector over a sparse Pauli operator. The operation is,
 *
 * S = \sum_i w_i <psi | \hat S_i | psi > 
 *
 * Where { \hat S_i } are the Pauli strings of the input pauli, { w_i } are the
 * weights of the input pauli, and psi is the statevector obtained by running
 * the input circuit on the input statevector_h. 
 *
 * This operation requires slightly more memory than run_pauli_expectation, but
 * requires many fewer dot iterations. Moreover, this operation is highly
 * similar in implementation to run_pauli_expectation_value_gradient below.
 *
 * GPU memory:
 *  3x statevector
 *
 * Host-device transfers:
 *  1x statevector if input statevector_h is provided 
 *  Negligible if input statevector_h is NULL
 *
 * Params:
 *  circuit (Circuit) - circuit to run
 *  pauli (Pauli) - pauli operator indicating operator to evaluate expectation
 *      value for
 *  statevector_h (T*) - input statevector or NULL to indicate the |000...>
 *      statevector
 *  compressed (bool) - apply compression to the circuit (true) or not (false)?
 * Returns: 
 *  (T) - the scalar expectation value over the sparse Pauli operator.
 **/
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

    T* statevector1_d = vulcan::gpu::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector1_d);
    vulcan::util::apply_pauli(pauli, statevector1_d, statevector2_d, statevector3_d);
    T val = vulcan::gpu::dot(circuit.nqubit(), statevector1_d, statevector2_d);
    vulcan::gpu::free_statevector(statevector1_d);
    vulcan::gpu::free_statevector(statevector2_d);
    vulcan::gpu::free_statevector(statevector3_d);
    return val;
}
    
/**
 * Run a quantum circuit, evaluate the expectation value of the resultant
 * statevector over a sparse Pauli operator, and then evaluate the gradient of
 * this expectation value with respect to the "theta" angle parameters of all
 * 1-qubit Rx/Ry/Rz gates in the original quantum circuit. The operation is,
 *
 * { d S / d theta_g } = (d / d theta_g) 
 *  \sum_i w_i <psi ({ theta_g }) | \hat S_i | psi ({ theta_g })> 
 *
 * Where { \hat S_i } are the Pauli strings of the input pauli, { w_i } are the
 * weights of the input pauli, and psi is the statevector obtained by running
 * the input circuit on the input statevector_h. { theta_g } are the set of
 * input angles to the Rx/Ry/Rz gates of the input circuit. 
 *
 * Note that the R_G angles are defined in full-angle convention, e.g., R_G =
 * e^{-i theta G}. 
 *
 * Note that only the 1-qubit size and Rx/Ry/Rz name fields of the gates are
 * checked to determine which gates contain derivative generators. The specific
 * matrix values of the gates are not checked.
 *
 * Note that the method is built to return correctly in real types for real
 * statevectors/operators, i.e., if the statevector is real-valued, contains
 * only Ry parametrized gates (no Rx/Rz gates), and is evaluated over a real,
 * symmetric Pauli operator.
 *
 * GPU memory:
 *  3x statevector
 *
 * Host-device transfers:
 *  1x statevector if input statevector_h is provided 
 *  Negligible if input statevector_h is NULL
 *
 * Params:
 *  circuit (Circuit) - circuit to run
 *  pauli (Pauli) - pauli operator indicating operator to evaluate expectation
 *      value for
 *  statevector_h (T*) - input statevector or NULL to indicate the |000...>
 *      statevector
 *  compressed (bool) - apply compression to the circuit (true) or not (false)?
 * Returns: 
 *  (std::vector<T>) - the gradient of the scalar Pauli expectation value with
 *      respect to the theta angles of all 1-qubit Rx/Ry/Rz gates encountered
 *      in the circuit, returned in time-ordered sorting
 **/
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

    T* statevector1_d = vulcan::gpu::malloc_and_initialize_statevector(pauli.nqubit(), statevector_h);
    T* statevector2_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
    T* statevector3_d = vulcan::gpu::malloc_statevector<T>(pauli.nqubit());
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
            gradient.push_back(T(2.0 * vulcan::gpu::dot(circuit.nqubit(), statevector1_d, statevector3_d).real()));
        }
    }

    std::reverse(gradient.begin(), gradient.end());

    vulcan::gpu::free_statevector(statevector1_d);
    vulcan::gpu::free_statevector(statevector2_d);
    vulcan::gpu::free_statevector(statevector3_d);
    return gradient;
}

/**
 * Run a quantum circuit and then sample random measurements (kets) from the
 * final statevector.
 *
 * Note that if an input statevector_h is not provided, this is an elegant
 * "quantum" operation: The circuit data is sent to the GPU, the GPU
 * initializes the quantum statevector in the |000...> state, runs the circuit,
 * performs random measurements, and returns these measurements to the host. No
 * 2**nqubit-sized data crosses the host-device bus, and the operation directly
 * emulates the interface with a real quantum device.
 *
 * Specifically, the GPU call sequence is:
 *  run_statevector -> abs2 -> cumsum -> measure
 *
 * See these specific functions in vulcan_gpu.hpp for details.
 *
 * GPU memory:
 *  1x statevector if T and U are the same real type
 *  1.5x statevector if T is complex and U is real
 *
 * Host-device transfers:
 *  1x statevector if input statevector_h is provided 
 *  Negligible if input statevector_h is NULL
 *
 * The statevector is run in type T, the abs2, cumsum, and measure operations
 * are run in type U (always a real type).
 *
 * The following combinations of types are instantiated:
 *   run_measurement<float32, float32> 
 *   run_measurement<float64, float64> 
 *   run_measurement<complex64, float32> 
 *   run_measurement<complex128, float64> 
 *
 * Params:
 *  circuit (Circuit) - circuit to run
 *  statevector_h (T*) - input statevector or NULL to indicate the |000...>
 *      statevector
 *  nmeasurement (int) - number of measurements to run
 *  randoms (U*) - input real random uniform samples in [0, 1). Size
 *      nmeasurement
 *  measurements_h (int*) - host register to place output measurements into.
 *      Size nmeasurement
 *  compressed (bool) - apply compression to the circuit (true) or not (false)?
 * Result:
 *  measurements_h is overwritten with the randomly sampled kets
 **/
template <typename T, typename U>
void run_measurement(
    const Circuit<T>& circuit,
    const T* statevector_h,
    int nmeasurement,
    const U* randoms_h,
    int* measurements_h,
    bool compressed)
{
    T* statevector_d = vulcan::gpu::malloc_and_initialize_statevector(circuit.nqubit(), statevector_h);
    vulcan::util::run_statevector(compressed ? circuit.compressed() : circuit, statevector_d);

    U* pvector_d;
    if (sizeof(T) == sizeof(U)) {
        pvector_d = (U*) statevector_d; // in-place with real types
    } else if (sizeof(T) == 2*sizeof(U)) {
        pvector_d = vulcan::gpu::malloc_statevector<U>(circuit.nqubit()); // Temporary with complex types
    } else {
        throw std::runtime_error("T/U sizes do not make sense");
    }

    vulcan::gpu::abs2<T, U>(circuit.nqubit(), statevector_d, pvector_d);

    U sum = vulcan::gpu::cumsum<U>(circuit.nqubit(), pvector_d);

    U* randoms_d;
    cudaMalloc(&randoms_d, sizeof(U)*nmeasurement);
    cudaMemcpy(randoms_d, randoms_h, sizeof(U)*nmeasurement, cudaMemcpyHostToDevice);

    int* measurements_d;
    cudaMalloc(&measurements_d, sizeof(int)*nmeasurement);

    vulcan::gpu::measure(circuit.nqubit(), pvector_d, sum, nmeasurement, randoms_d, measurements_d); 

    cudaMemcpy(measurements_h, measurements_d, sizeof(int)*nmeasurement, cudaMemcpyDeviceToHost);

    vulcan::gpu::free_statevector(statevector_d);
    if (sizeof(T) != sizeof(U)) {
        vulcan::gpu::free_statevector(pvector_d);
    }
    cudaFree(randoms_d);
    cudaFree(measurements_d);
}
    
} // namespace vulcan
