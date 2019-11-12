import quasar
import vulcan
import time
import numpy as np
import sys

class Configuration(object):

    def __init__(
        self,
        backend,
        dtype,
        threshold,
        ):

        self.backend = backend
        self.dtype = dtype
        self.threshold = threshold

def test_configuration(
    circuit,
    pauli,
    tasks,
    configuration,
    ):

    values = {}
    times = {}

    backend = configuration.backend
    dtype = configuration.dtype

    # Warm CUDA Up
    backend.run_statevector(circuit, dtype=dtype)

    if 'run_statevector' in tasks:
        start = time.time()
        values['run_statevector'] = backend.run_statevector(circuit, dtype=dtype)
        times['run_statevector'] = time.time() - start

    if 'run_pauli_sigma' in tasks:
        start = time.time()
        values['run_pauli_sigma'] = backend.run_pauli_sigma(pauli, values['run_statevector'], dtype=dtype)
        times['run_pauli_sigma'] = time.time() - start

    if 'run_pauli_expectation' in tasks:
        start = time.time()
        values['run_pauli_expectation'] = np.array(backend.run_pauli_expectation(circuit, pauli, dtype=dtype).values(), dtype=dtype)
        times['run_pauli_expectation'] = time.time() - start

    if 'run_pauli_expectation_value' in tasks:
        start = time.time()
        values['run_pauli_expectation_value'] = backend.run_pauli_expectation_value(circuit, pauli, dtype=dtype)
        times['run_pauli_expectation_value'] = time.time() - start

    if 'run_pauli_expectation_value_gradient' in tasks:
        start = time.time()
        values['run_pauli_expectation_value_gradient'] = backend.run_pauli_expectation_value_gradient(circuit, pauli, dtype=dtype)
        times['run_pauli_expectation_value_gradient'] = time.time() - start

    return values, times

def test_array(
    circuit,
    pauli,
    tasks,
    configurations,
    reference,
    ):
    
    values = {}
    times = {}

    for key, configuration in configurations.items():
        values[key], times[key] = test_configuration(circuit, pauli, tasks, configuration)

    deltas = {}
    for key in configurations.keys():
        deltas[key] = { key2 : np.max(np.abs(value - values[reference][key2])) for key2, value in values[key].items() }

    for task in tasks:
        print('Task: %s' % (task))
        print('')

        for key in configurations.keys():
            print('%30s : %11.3E %11.3E %11.3E' % (
                key,
                times[key][task],
                deltas[key][task],
                times[key][task] / times[reference][task],
                ))
            

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


def test_cis_z1(
    nqubit,
    ):

    circuit = cis_circuit(nqubit)
    pauli = z1_pauli(nqubit)

    tasks = [
        'run_statevector',
        'run_pauli_sigma',  
        'run_pauli_expectation',  
        'run_pauli_expectation_value',  
    #     'run_pauli_expectation_value_gradient',  
    ]

    backends = {
    #     'quasar_slow' : quasar.QuasarSimulatorBackend(),
    #     'quasar' : quasar.QuasarUltrafastBackend(),
        'vulcan' : vulcan.VulcanSimulatorBackend(),
    #     'qiskit' : quasar.QiskitSimulatorBackend(),
        'cirq' : quasar.CirqSimulatorBackend(),
    }

    dtypes = {
        'float32' : np.float32,
    #     'float64' : np.float64,
        'complex64' : np.complex64,
    #     'complex128' : np.complex128,
    }

    configurations = {}
    for key, backend in backends.items():
        for key2, dtype in dtypes.items():  
            configurations[key + '_' + key2] = Configuration(
                backend=backend,
                dtype=dtype,
                threshold=1.0E-5 if key2 in ('float32', 'complex64') else 1.0E-12,
                )
    # del configurations['qiskit_float32']
    # del configurations['qiskit_complex64']
    # del configurations['qiskit_float64']
    # del configurations['qiskit_complex128']

    configurations = {
        'vulcan_float32' : Configuration(backend=backends['vulcan'], dtype=dtypes['float32'], threshold=1.0E-5),
        'cirq_complex64' : Configuration(backend=backends['cirq'], dtype=dtypes['complex64'], threshold=1.0E-5),
    }

    test_array(
        circuit=circuit,
        pauli=pauli,
        tasks=tasks,
        configurations=configurations,
        reference='vulcan_float32',
        )


if __name__ == '__main__':

    nqubit = int(sys.argv[1])

    test_cis_z1(nqubit)
    
