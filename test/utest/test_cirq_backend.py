import quasar
import numpy as np
from util.error import L1_error
from util.circuit_generator import random_circuit

"""
Test "CirqBackend" Class
"""

def angle():
    """
    Validate quasar_to_cirq_angle() and cirq_to_quasar_angle() of the "CirqBackend" class. 
    """
    backend = quasar.CirqBackend()
    angle1 = backend.quasar_to_cirq_angle(10)
    
    backend = quasar.CirqBackend()
    angle2 = backend.cirq_to_quasar_angle(10)

    return angle1==20 and angle2==5


def build_native_circuit():
    """
    Validate build_native_circuit() of the "CirqBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    cirq_backend = quasar.CirqBackend()
    cirq_circuit = cirq_backend.build_native_circuit(quasar_circuit)
    # directly build a cirq circuit
    import cirq
    q = [cirq.LineQubit(A) for A in range(2)]
    bell = cirq.Circuit()
    bell.append(cirq.H(q[0]))
    bell.append(cirq.CNOT(q[0],q[1]))
    
    return cirq_circuit==bell


def build_quasar_circuit():
    """
    Validate build_quasar_circuit() of the "CirqBackend" class. 
    """
    # translate circuit from cirq circuit
    import cirq
    q = [cirq.LineQubit(A) for A in range(2)]
    cirq_circuit = cirq.Circuit()
    cirq_circuit.append(cirq.H(q[0]))
    cirq_circuit.append(cirq.CNOT(q[0],q[1]))
    cirq_backend = quasar.CirqBackend()
    quasar_circuit = cirq_backend.build_quasar_circuit(cirq_circuit)
    
    # directly build a quasar circuit
    bell = quasar.Circuit(N=2).H(0).CX(0,1)

    return quasar_circuit.is_equivalent(bell)

    
def build_native_circuit_in_basis():
    """
    Validate build_native_circuit_in_basis() of the "CirqBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    cirq_backend = quasar.CirqBackend()
    cirq_circuit = cirq_backend.build_native_circuit_in_basis(quasar_circuit,['X','Z'])
    # directly build a cirq circuit
    # directly build a cirq circuit
    import cirq
    q = [cirq.LineQubit(A) for A in range(2)]
    bell = cirq.Circuit()
    bell.append(cirq.H(q[0]))
    bell.append(cirq.CNOT(q[0],q[1]))
    bell.append(cirq.H(q[0]))
    
    return cirq_circuit==bell


def build_native_circuit_measurement():
    """
    Validate build_native_circuit_measurement() of the "CirqBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    cirq_backend = quasar.CirqBackend()
    cirq_circuit = cirq_backend.build_native_circuit_measurement(quasar_circuit)
    # directly build a cirq circuit
    import cirq
    q = [cirq.LineQubit(A) for A in range(2)]
    bell = cirq.Circuit()
    bell.append(cirq.H(q[0]))
    bell.append(cirq.CNOT(q[0],q[1]))
    for qubit in bell.all_qubits():
            bell.append(cirq.measure(qubit))
    
    return cirq_circuit==bell


"""
Test "CirqSimulatorBackend" Class
"""
def util_build_circuit():
    """
    A utility function for the testing functions below.
    It creates a quasar Circuit object.
    """
    circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    
    return circuit


def cirq_simulator_backend():
    """
    Validate initialization of the "CirqSimulatorBackend" class.
    """
    backend = quasar.CirqSimulatorBackend()
    
    if not backend.has_statevector: return False
    if not backend.has_measurement: return False
    
    return True

    
def run_statevector():
    """
    Validate run_statevector() in the "CirqSimulatorBackend" class.
    """
    backend = quasar.CirqSimulatorBackend()
    # test case 1: Bell's state
    circuit = util_build_circuit()
    result = backend.run_statevector(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    # test case 2: Random circuit
    circuit = random_circuit(seed=5566, depth=10)
    result = backend.run_statevector(circuit)
    ans = circuit.simulate()
    
    return L1_error((result, ans))   
    
   
def run_measurement():
    """
    Validate run_measurement() in the "CirqSimulatorBackend" class.
    """
    backend = quasar.CirqSimulatorBackend()
    circuit = util_build_circuit()
    result = backend.run_measurement(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    
    # these two lines normalize the result
    result = [result.get('{0:02b}'.format(k), 0) for k in range(4)]
    result = np.array(result)/np.linalg.norm(result)
    
    return L1_error((result, ans), margin=0.2)


"""
Test "cirqHardwareBackend" Class
Hardware test not implemented yet
"""

# def cirq_hardware_backend():
    # """
    # Validate initialization of the "cirqHardwareBackend" class.
    # """
    # backend = quasar.cirqHardwareBackend()
    
    # if backend.has_statevector: return False
    # if not backend.has_measurement: return False
    
    # return True
    
    








