#pragma once

#include <stdexcept>
#include <vector>

namespace vulcan {    

template<typename T>
class Gate;

template<typename T>
class Circuit;

template<typename T>
class Pauli;

namespace util {

// => 2x2 and 4x4 Linear Algebra Utility <= //

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

// => Internal C++ API <= //

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
            vulcan::gpu::apply_gate_1(
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
            vulcan::gpu::apply_gate_2(
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

    vulcan::gpu::apply_gate_1(
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

    vulcan::gpu::apply_gate_1(
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

    vulcan::gpu::apply_gate_1(
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
    vulcan::gpu::zero_statevector(pauli.nqubit(), statevector2_d);
    for (size_t index = 0; index < pauli.types().size(); index++) {
        const std::vector<int>& types = pauli.types()[index];
        const std::vector<int>& qubits = pauli.qubits()[index];
        T val = pauli.values()[index];
        vulcan::gpu::axpby(pauli.nqubit(), statevector1_d, statevector3_d);
        int ny = 0;
        for (size_t index2 = 0; index2 < types.size(); index2++) {
            int type = types[index2];
            int qubit = qubits[index2];          
            if (type == 1) ny += 1;
            vulcan::util::apply_pauli_1_mody(pauli.nqubit(), statevector3_d, statevector3_d, qubit, type);
        }  
        // (-i)**ny
        T scal = mi_pow_n<T>(ny);
        vulcan::gpu::axpby(pauli.nqubit(), statevector3_d, statevector2_d, scal*val, T(1.0));
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
        vulcan::gpu::axpby(pauli.nqubit(), statevector1_d, statevector2_d);
        int ny = 0;
        for (size_t index2 = 0; index2 < types.size(); index2++) {
            int type = types[index2];
            int qubit = qubits[index2];          
            if (type == 1) ny += 1;
            vulcan::util::apply_pauli_1_mody(pauli.nqubit(), statevector2_d, statevector2_d, qubit, type);
        }  
        // (-i)**ny
        T scal = mi_pow_n<T>(ny);
        values[index] = scal*vulcan::gpu::dot(pauli.nqubit(), statevector1_d, statevector2_d); 
    }
    
    return Pauli<T>(
        pauli.nqubit(),
        pauli.types(),
        pauli.qubits(),
        values);
}

} // namespace vulcan::util
    
} // namespace vulcan
