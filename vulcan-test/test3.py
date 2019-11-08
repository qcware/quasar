import quasar
import vulcan
import time
import numpy as np
import sys

def test_circuit(
    backend,
    circuit,
    pauli,
    dtype,
    ):

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

def test_pair(
    backend1,
    backend2, 
    circuit,
    pauli,
    dtype1,
    dtype2,
    threshold,
    ):

    values1, times1 = test_circuit(
        backend=backend1,
        circuit=circuit,
        pauli=pauli,
        dtype=dtype1,
        )

    values2, times2 = test_circuit(
        backend=backend2,
        circuit=circuit,
        pauli=pauli,
        dtype=dtype2,
        )
    print(circuit)

    print('%-36s : %12s %12s %12s %3s' % (
        'Key', 
        'T (backend1)',
        'T (backend2)',
        'Deviation',
        'OK',
        ))
    
    OK = True
    for key in values2.keys():
        delta = np.max(np.abs(values1[key] - values2[key]))
        valid = delta < threshold
        print('%-36s : %12.3E %12.3E %12.3E %3s' % (
            key,
            times1[key],
            times2[key],
            delta,
            'OK' if valid else 'BAD',
            ))
        OK &= valid

    print('OK = %s' % (
        'OK' if OK else 'BAD',
        ))

    return OK

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

def cis_t_circuit(
    nqubit=4,
    ):

    gadget = quasar.Circuit().Ry(1).CZ(0,1).Ry(1).CX(1,0)
    
    circuit = quasar.Circuit().X(0)
    circuit.T(0)
    for I in range(nqubit-1):
        circuit.add_gates(circuit=gadget, qubits=(I, I+1))
        circuit.T(I)
    
    parameter_values = []
    for I in range(nqubit - 1):
        value = (1.0 - I / 17.5)
        parameter_values.append(+value)
        parameter_values.append(-value)
    circuit.set_parameter_values(parameter_values)
    
    return circuit

def x1_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        pauli += (k + 1) / 10.0 * X[k]
    
    return pauli

def y1_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        pauli += (k + 1) / 10.0 * Y[k]
    
    return pauli

def z1_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        pauli += (k + 1) / 10.0 * Z[k]
    
    return pauli

def y2_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        if k % 2 != 0: continue
        pauli += (k + 1) / 10.0 * Y[k] * Y[k+1]
    
    return pauli

def y3_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        if k % 3 != 0: continue
        pauli += (k + 1) / 10.0 * Y[k] * Y[k+1] * Y[k+2]
    
    return pauli

def y4_pauli(
    nqubit=4,
    ):

    I, X, Y, Z = quasar.Pauli.IXYZ()
    pauli = quasar.Pauli.zero()
    for k in range(nqubit):
        if k % 4 != 0: continue
        pauli += (k + 1) / 10.0 * Y[k] * Y[k+1] * Y[k+2] * Y[k+3]
    
    return pauli

    

if __name__ == '__main__':

    nqubit = int(sys.argv[1])
    
    dtype1 = {
        'float32' : np.float32,
        'float64' : np.float64,
        'complex64' : np.complex64,
        'complex128' : np.complex128,
    }[sys.argv[2]]

    dtype2 = {
        'float32' : np.float32,
        'float64' : np.float64,
        'complex64' : np.complex64,
        'complex128' : np.complex128,
    }[sys.argv[3]]

    backends = {
        'quasar_slow' : quasar.QuasarSimulatorBackend(),
        'quasar' : quasar.QuasarUltrafastBackend(),
        'vulcan' : vulcan.VulcanSimulatorBackend(),
    }

    backend1 = backends[sys.argv[4]]
    backend2 = backends[sys.argv[5]]

    threshold = float(sys.argv[6])

    circuits = {
        'cis' : cis_circuit,
        'cis_t' : cis_t_circuit,
    }
    circuit = circuits[sys.argv[7]](nqubit)

    paulis = {
        'x1' : x1_pauli,
        'y1' : y1_pauli,
        'z1' : z1_pauli,
        'y2' : y2_pauli,
        'y3' : y3_pauli,
        'y4' : y4_pauli,
    }
    pauli = paulis[sys.argv[8]](nqubit)

    test_pair(
        backend1=backend1,
        backend2=backend2,
        circuit=circuit,
        pauli=pauli,
        dtype1=dtype1,
        dtype2=dtype2,
        threshold=threshold,
        )

