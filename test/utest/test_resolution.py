import quasar
import numpy as np


def build_quasar_circuit_from_quasar():
    native_circuit = quasar.Circuit(N=2)
    quasar_circuit = quasar.build_quasar_circuit(native_circuit)
    
    return isinstance(quasar_circuit, quasar.Circuit)
    
    
def build_quasar_circuit_from_cirq():
    # build a testing native circuit
    import cirq 
    native_circuit = cirq.Circuit()
    native_circuit.append(cirq.I(cirq.LineQubit(0)))
    # translate to quasar
    quasar_circuit = quasar.build_quasar_circuit(native_circuit)
    
    return isinstance(quasar_circuit, quasar.Circuit)    
    
    
def build_quasar_circuit_from_qiskit():
    # build a testing native circuit
    import qiskit
    q = qiskit.QuantumRegister(1)
    native_circuit = qiskit.QuantumCircuit(q)
    # translate to quasar
    quasar_circuit = quasar.build_quasar_circuit(native_circuit)
    
    return isinstance(quasar_circuit, quasar.Circuit)     
    
    
def build_quasar_circuit_from_forest():
    # build a testing native circuit
    import pyquil
    native_circuit = pyquil.Program(pyquil.gates.I(0))
    # translate to quasar
    quasar_circuit = quasar.build_quasar_circuit(native_circuit)
    
    return isinstance(quasar_circuit, quasar.Circuit)

    















