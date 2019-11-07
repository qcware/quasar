import quasar
import vulcan
import time
import numpy as np
import sys

def test_cis_circuit(
    nqubit=4,
    dtype=np.complex128,
    backend=vulcan.VulcanSimulatorBackend()
    ):

    gadget = quasar.Circuit().Ry(1).CZ(0,1).Ry(1).CX(1,0)
    
    circuit = quasar.Circuit().X(0)
    for I in range(nqubit-1):
        circuit.add_gates(circuit=gadget, qubits=(I, I+1))
    print(circuit)
    
    parameter_values = []
    for I in range(nqubit - 1):
        value = (1.0 - I / 17.5)
        parameter_values.append(+value)
        parameter_values.append(-value)
    circuit.set_parameter_values(parameter_values)
    print(circuit.parameter_str)
    
    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        pauli += (k + 1) / 10.0 * Z[k]
    print(pauli)

    values = {}
    times = {}

    # Warm CUDA Up
    backend.run_statevector(circuit, dtype=dtype)

    start = time.time()
    values['run_statevector'] = backend.run_statevector(circuit, dtype=dtype)
    times['run_statevector'] = time.time() - start

    start = time.time()
    values['run_pauli_sigma'] = backend.run_pauli_sigma(pauli, values['run_statevector'], dtype=dtype)
    times['run_pauli_sigma'] = time.time() - start

    start = time.time()
    values['run_pauli_expectation_value'] = backend.run_pauli_expectation_value(circuit, pauli, dtype=dtype)
    times['run_pauli_expectation_value'] = time.time() - start

    start = time.time()
    values['run_pauli_expectation_value_gradient'] = backend.run_pauli_expectation_value_gradient(circuit, pauli, dtype=dtype)
    times['run_pauli_expectation_value_gradient'] = time.time() - start

    return values, times

if __name__ == '__main__':

    nqubit = int(sys.argv[1])
    dtype = {
        'float32' : np.float32,
        'float64' : np.float64,
        'complex64' : np.complex64,
        'complex128' : np.complex128,
    }[sys.argv[2]]

    # backend1 = quasar.QuasarSimulatorBackend()
    # backend1 = quasar.QuasarUltrafastBackend()
    backend1 = vulcan.VulcanSimulatorBackend()
    backend2 = vulcan.VulcanSimulatorBackend()

    values1, times1 = test_cis_circuit(
        nqubit=nqubit,
        dtype=dtype,
        backend=backend1,
        )

    values2, times2 = test_cis_circuit(
        nqubit=nqubit,
        dtype=dtype,
        backend=backend2,
        )

    print('%-36s : %11s %11s %11s' % (
        'Key', 
        'T (quasar)',
        'T (vulcan)',
        'Deviation',
        ))
    for key in values2.keys():
        print('%-36s : %11.3E %11.3E %11.3E' % (
            key,
            times1[key],
            times2[key],
            np.max(np.abs(values1[key] - values2[key])),
            ))
    print(values1['run_pauli_expectation_value_gradient'])
