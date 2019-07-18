import quasar
import numpy as np
from util.error import L1_error

"""
Test "build_native_circuit" for all 4 backends.
"""

def build_native_circuit_to_quasar():
    """
    Validate "build_quasar_circuit()"
    """
    # build a testing quasar circuit
    quasar_circuit = quasar.Circuit(N=1).I(0)
    # translate to native circuit
    backend = quasar.QuasarSimulatorBackend()
    native_circuit = quasar.build_native_circuit(backend, quasar_circuit)
    
    return isinstance(quasar_circuit, quasar.Circuit)    
    

def build_native_circuit_to_cirq():
    """
    Validate "build_native_circuit()"
    """
    # build a testing quasar circuit
    quasar_circuit = quasar.Circuit(N=1).I(0)
    # translate to native circuit
    import cirq
    backend = quasar.CirqBackend()
    native_circuit = quasar.build_native_circuit(backend,quasar_circuit)
    
    return isinstance(native_circuit, cirq.Circuit)    

    
def build_native_circuit_to_qiskit():
    """
    Validate "build_native_circuit()"
    """
    # build a testing quasar circuit
    quasar_circuit = quasar.Circuit(N=1).I(0)
    # translate to native circuit
    import qiskit
    backend = quasar.QiskitBackend()
    native_circuit = quasar.build_native_circuit(backend,quasar_circuit)
    
    return isinstance(native_circuit, qiskit.QuantumCircuit)    
    
    
def build_native_circuit_to_forest():
    """
    Validate "build_native_circuit()"
    """
    # build a testing quasar circuit
    quasar_circuit = quasar.Circuit(N=1).I(0)
    # translate to native circuit
    import pyquil
    backend = quasar.ForestBackend()
    native_circuit = quasar.build_native_circuit(backend,quasar_circuit)
    
    return isinstance(native_circuit, pyquil.Program)    
    
    
"""
Execuate "run_measurement" and "run_statevector", then compare the results with the quasar's ones. 
"""

def util_backends():
    quasar_backend = quasar.QuasarSimulatorBackend()
    qiskit_backend = quasar.QiskitSimulatorBackend()
    cirq_backend = quasar.CirqSimulatorBackend()
    #forest_backend = quasar.ForestSimulatorBackend('2q-qvm')
    #return [quasar_backend, qiskit_backend, cirq_backend, forest_backend]
    return [quasar_backend, qiskit_backend, cirq_backend]
    
def util_circuits():
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    qiskit_circuit = quasar.QiskitSimulatorBackend().build_native_circuit(quasar_circuit)
    cirq_circuit   = quasar.CirqSimulatorBackend().build_native_circuit(quasar_circuit)
    #forest_circuit = quasar.ForestSimulatorBackend('2q-qvm').build_native_circuit(quasar_circuit)
    # return [quasar_circuit, qiskit_circuit, cirq_circuit, forest_circuit]
    return [quasar_circuit, qiskit_circuit, cirq_circuit]    

def run_measurement():
    backends = util_backends()
    circuits = util_circuits()
    nmeasurement = 1000
    
    results = []
    for backend in backends:
        for circuit in circuits:
            results.append(quasar.run_measurement(backend, circuit,nmeasurement=nmeasurement))
    results = [(list(result.values()),list(results[0].values())) for result in results]
    return L1_error(results, margin=1.5*nmeasurement)
    

def run_statevector():
    backends = util_backends()
    circuits = util_circuits()
    
    results = []
    for backend in backends:
        for circuit in circuits:
            results.append(quasar.run_statevector(backend, circuit))
    results = [(result,results[0]) for result in results]
    return L1_error(results)
    

def run_pauli_expectation():
    backends = util_backends()
    circuits = util_circuits()
    
    I,X,Y,Z = quasar.Pauli.IXYZ()
    pauli = Z[0] * Z[1]
    
    results = []
    for backend in backends:
        for circuit in circuits:
            results.append(quasar.run_pauli_expectation(backend, circuit, pauli))
    
    results = [list(r.values())[0] for r in results]
    results = [(result,results[0]) for result in results]
    return L1_error(results)


def run_unitary():
    backends = util_backends()
    circuits = util_circuits()
    
    results = []
    for backend in backends:
        for circuit in circuits:
            results.append(quasar.run_unitary(backend, circuit))
    results = [(result,results[0]) for result in results]
    return L1_error(results)    

    
def run_density_matrix():
    backends = util_backends()
    circuits = util_circuits()
    
    results = []
    for backend in backends:
        for circuit in circuits:
            results.append(quasar.run_unitary(backend, circuit))
    results = [(result,results[0]) for result in results]
    return L1_error(results) 
    
