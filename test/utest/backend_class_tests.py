"""
Tests for the Backend class in circuit.py
"""

import quasar
import numpy as np
import collections

def run_pauli_expectation():
    # this just calls either run from statevector or run from measurement
    pass

def run_unitary():
    # gets the unitary matrix that represents the circuit
    back = quasar.Backend()
    circuit = quasar.Circuit(1).X(0).Z(0)
    unitary = back.run_unitary(circuit)
    unitary1 = back.run_unitary(circuit, compressed=False)

    # Checking compressed
    if np.allclose(unitary, unitary1):
        return True
    else:
        return False

def run_unitary1():
    # gets the unitary matrix that represents the circuit
    back = quasar.Backend()
    circuit = quasar.Circuit(1).X(0).Z(0)
    unitary = back.run_unitary(circuit)

    u = np.array([
                [0, 1],
                [-1, 0]])
    
    # Checking unitary is correct
    if np.allclose(unitary, u):
        return True
    else:
        return False

def run_density_matrix():
    back = quasar.Backend()
    circuit = quasar.Circuit(1).X(0)
    dm = back.run_density_matrix(circuit)
    
    one_dm = np.array([
                        [0,0],
                        [0,1]])

    if np.allclose(dm, one_dm):
        return True
    else:
        return False

def run_density_matrix1():
    back = quasar.Backend()
    circuit = quasar.Circuit(1).X(0)
    starter_wfn = np.array([0,1])
    dm = back.run_density_matrix(circuit, starter_wfn)
    
    zero_dm = np.array([
                        [1,0],
                        [0,0]])

    if np.allclose(dm, zero_dm):
        return True
    else:
        return False

def run_density_matrix_compressed():
    back = quasar.Backend()
    circuit = quasar.Circuit(2).X(0).Z(1).CX(0,1)
    dm = back.run_density_matrix(circuit)
    dm1 = back.run_density_matrix(circuit, compressed=False)
    
    if np.allclose(dm, dm1):
        return True
    else:
        return False

def run_pauli_expectation_from_statevector():
    qback = quasar.QuasarSimulatorBackend()
    circuit = quasar.Circuit(1).X(0)
    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = Z[0]
    pe = qback.run_pauli_expectation_from_statevector(circuit, pauli)
    
    if len(pe.values()) != 1:
        return False
    for v in pe.values():
        if v != -1:
            return False
    
    return True

def run_pauli_expectation_from_statevector1():
    # *** Need to check if this is right before adding it to unittest file (backend_test.py) ***

    qback = quasar.QuasarSimulatorBackend()
    circuit = quasar.Circuit(1).H(0).Rz(0, theta=np.pi/8)
    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = 0.5*X[0] + Y[0] # would expect expectation for the X term to be diminished

    # Shouldn't this result in a pauli expectaion of i if <0|XY|0> ?
    #circuit = quasar.Circuit(1).I(0)
    #pauli = X[0]*Y[0]

    #pauli = X[0]*X[0] + X[0] # Identity + X
    #pauli = X[0]*X[0] # Identity
    #pauli = I 
    
    pe = qback.run_pauli_expectation_from_statevector(circuit, pauli)
    
    print(X[0].values())
    print(pe)

    if len(pe.values()) != 1:
        return False
    for v in pe.values():
        if v != 1j:
            return False
    
    return True

def run_pauli_expectation_from_measurment():
    qback = quasar.QuasarSimulatorBackend()
    I, X, Y, Z = quasar.Pauli.IXYZ()
    circuit = quasar.Circuit(1).X(0)
    pauli = Z[0] # would expect expectation for the X term to be diminished

    pe = qback.run_pauli_expectation_from_measurement(circuit, pauli, nmeasurement=1000)

    if len(pe.values()) != 1:
        return False
    for v in pe.values():
        if v != -1:
            return False
    
    return True

def run_pauli_expectation_from_measurement1():
    # *** Need to check if this is right before adding it to unittest file (backend_test.py) ***

    qback = quasar.QuasarSimulatorBackend()
    I, X, Y, Z = quasar.Pauli.IXYZ()

    #circuit = quasar.Circuit(1).H(0).Rz(0, theta=np.pi/8)
    #pauli = 0.5*X[0] + Y[0] # would expect expectation for the X term to be diminished

    """
    # Not sure what should be returned here, because not sure how to decomposition works
    sv = circuit.simulate()
    H = np.array([[0, 0.5-1j],
                  [0.5+1j, 0]])
    step1 = np.dot(H, sv)
    step2 = np.dot(sv.conj(), step1)
    print(step2)
    """

    # Shouldn't this result in a pauli expectaion of i if <0|XY|0> ?
    circuit = quasar.Circuit(1).I(0)
    pauli = X[0]*Y[0]

    # Doensn't like identity
    #pauli = I

    pe = qback.run_pauli_expectation_from_statevector(circuit, pauli)
    
    print(X[0].values())
    print(pe)

    if len(pe.values()) != 1:
        return False
    for v in pe.values():
        if v != 1j:
            return False
    
    return True

def bit_reversal_permutation():
    b = quasar.Backend()

    for i in range(5):
        result = b.bit_reversal_permutation(i) 
        if len(result) != 2**i:
            return False
        for i in range(len(result)//2):
            max_idx = len(result)-1
            if result[i] + result[max_idx - i] != max_idx:
                return False

    return True

def statevector_bit_reversal_permutation():
    b = quasar.Backend()
    circuit = quasar.Circuit(2).X(1)
    statevector = circuit.simulate()

    result = b.statevector_bit_reversal_permutation(statevector)
    should_be = np.array([0, 0, 1, 0])
    
    if np.array_equal(result.real, should_be):
        return True
    else:
        return False
    
def statevector_bit_reversal_permutation1():
    b = quasar.Backend()

    circ2 = quasar.Circuit(2).X(0).H(0).CX(0,1).H(0)
    statevector = circ2.simulate().round(3)

    result = b.statevector_bit_reversal_permutation(statevector)
    # should swap inner elements
    should_be = np.array([0.5, 0.5, -0.5, 0.5])

    if np.array_equal(result.real, should_be):
        return True
    else:
        return False

# linear_commuting_group, how to test this

if __name__ == "__main__":
    run_unitary()
    run_unitary1()
    run_density_matrix()
    run_density_matrix_compressed()
    run_density_matrix1()
    run_pauli_expectation_from_statevector()
    #print(run_pauli_expectation_from_statevector1())
    run_pauli_expectation_from_measurment()
    #print(run_pauli_expectation_from_measurement1())
    bit_reversal_permutation()
    statevector_bit_reversal_permutation()
    statevector_bit_reversal_permutation1()
    
    
