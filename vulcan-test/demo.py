import quasar
import vulcan
import time
import numpy as np

def cis_circuit(
    nqubit=4,
    ):

    gadget = quasar.Circuit().Ry(1).CZ(0,1).Ry(1).CX(1,0)
    
    circuit = quasar.Circuit().X(0)
    for I in range(nqubit-1):
        circuit.add_gates(circuit=gadget, qubits=(I, I+1))
    
    parameter_values = []
    for I in range(nqubit - 1):
        value = (1.0 - I / 17.5)
        parameter_values.append(+value)
        parameter_values.append(-value)
    circuit.set_parameter_values(parameter_values)
    
    return circuit

def z1_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        pauli += (k + 1) / 10.0 * Z[k]
    
    return pauli

if __name__ == '__main__':

    import sys
    nqubit = int(sys.argv[1])

    circuit = cis_circuit(nqubit)
    print(circuit)

    pauli = z1_pauli(nqubit)
    print(pauli)

    dtype = np.float32
    # dtype = np.complex64
    # backend = quasar.CirqSimulatorBackend()
    # backend = quasar.QuasarSimulatorBackend()
    backend = vulcan.VulcanSimulatorBackend()

    start = time.time()
    statevector = backend.run_statevector(circuit, dtype=dtype)
    print('%11.3E' % (time.time() - start))

    start = time.time()
    energy = backend.run_pauli_expectation_value(circuit, pauli, dtype=dtype)
    print('%11.3E' % (time.time() - start))
    
    start = time.time()
    gradient = backend.run_pauli_expectation_value_gradient(circuit, pauli, dtype=dtype)
    print('%11.3E' % (time.time() - start))

