import quasar
import quasar
import numpy as np
from util.error import L1_error

"""
Test "QiskitBackend" Class
"""

def angle():
    """
    Validate quasar_to_qiskit_angle() and qiskit_to_quasar_angle() of the "QiskitBackend" class. 
    """
    backend = quasar.QiskitBackend()
    angle1 = backend.quasar_to_qiskit_angle(10)
    
    backend = quasar.QiskitBackend()
    angle2 = backend.qiskit_to_quasar_angle(10)

    return angle1==20 and angle2==5


def build_native_circuit():
    """
    Validate build_native_circuit() of the "QiskitBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    qiskit_backend = quasar.QiskitBackend()
    qiskit_circuit = qiskit_backend.build_native_circuit(quasar_circuit)
    # directly build a qiskit circuit
    import qiskit
    q = qiskit_circuit.qregs[0]
    bell = qiskit.QuantumCircuit(q)
    bell.h(q[0])
    bell.cx(q[0],q[1])

    return qiskit_circuit==bell


def build_quasar_circuit():
    """
    Validate build_quasar_circuit() of the "QiskitBackend" class. 
    """
    # translate circuit from qiskit circuit
    import qiskit
    q = qiskit.QuantumRegister(2)
    qiskit_circuit = qiskit.QuantumCircuit(q)
    qiskit_circuit.h(q[0])
    qiskit_circuit.cx(q[0],q[1])
    qiskit_backend = quasar.QiskitBackend()
    quasar_circuit = qiskit_backend.build_quasar_circuit(qiskit_circuit)
    
    # directly build a quasar circuit
    bell = quasar.Circuit(N=2).H(0).CX(0,1)

    return quasar_circuit==bell

    
def build_quasar_circuit():
    """
    Validate build_quasar_circuit() of the "QiskitBackend" class. 
    """
    # translate circuit from qiskit circuit
    import qiskit
    q = qiskit.QuantumRegister(2)
    qiskit_circuit = qiskit.QuantumCircuit(q)
    qiskit_circuit.h(q[0])
    qiskit_circuit.cx(q[0],q[1])
    qiskit_backend = quasar.QiskitBackend()
    quasar_circuit = qiskit_backend.build_quasar_circuit(qiskit_circuit)
    
    # directly build a quasar circuit
    bell = quasar.Circuit(N=2).H(0).CX(0,1)

    return quasar_circuit==bell    
    
    
def build_native_circuit_in_basis():
    """
    Validate build_native_circuit_in_basis() of the "QiskitBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    qiskit_backend = quasar.QiskitBackend()
    qiskit_circuit = qiskit_backend.build_native_circuit_in_basis(quasar_circuit,['X','Z'])
    # directly build a qiskit circuit
    import qiskit
    q = qiskit_circuit.qregs[0]
    bell = qiskit.QuantumCircuit(q)
    bell.h(q[0])
    bell.cx(q[0],q[1])
    bell.h(q[0])
    
    return qiskit_circuit==bell


def build_native_circuit_measurement():
    """
    Validate build_native_circuit_measurement() of the "QiskitBackend" class. 
    """
    # translate circuit from quasar circuit
    quasar_circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    qiskit_backend = quasar.QiskitBackend()
    qiskit_circuit = qiskit_backend.build_native_circuit_measurement(quasar_circuit)
    # directly build a qiskit circuit
    import qiskit
    q = qiskit_circuit.qregs[0]
    c = qiskit_circuit.cregs[0]
    bell = qiskit.QuantumCircuit(q)
    bell.h(q[0])
    bell.cx(q[0],q[1])
    measure = qiskit.QuantumCircuit(q, c)
    measure.measure(q, c)
    bell = bell + measure

    return qiskit_circuit==bell


"""
Test "QiskitSimulatorBackend" Class
"""
def util_build_circuit():
    """
    A utility function for the testing functions below.
    It creates a quasar Circuit object.
    """
    circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    
    return circuit


def qiskit_simulator_backend():
    """
    Validate initialization of the "QiskitSimulatorBackend" class.
    """
    backend = quasar.QiskitSimulatorBackend()
    
    if not backend.has_statevector: return False
    if not backend.has_measurement: return False
    
    return True

    
def run_statevector():
    """
    Validate run_statevector() in the "QiskitSimulatorBackend" class.
    """
    backend = quasar.QiskitSimulatorBackend()
    circuit = util_build_circuit()
    result = backend.run_statevector(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    
    return L1_error((result, ans))   
    
    
def run_measurement():
    """
    Validate run_measurement() in the "QiskitSimulatorBackend" class.
    """
    backend = quasar.QiskitSimulatorBackend()
    circuit = util_build_circuit()
    result = backend.run_measurement(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    
    # these two lines normalize the result
    result = [result.get('{0:02b}'.format(k), 0) for k in range(4)]
    result = np.array(result)/np.linalg.norm(result)
    
    return L1_error((result, ans), margin=0.1)


"""
Test "QiskitHardwareBackend" Class
Hardware test not implemented yet
"""

# def qiskit_hardware_backend():
    # """
    # Validate initialization of the "QiskitHardwareBackend" class.
    # """
    # backend = quasar.QiskitHardwareBackend()
    
    # if backend.has_statevector: return False
    # if not backend.has_measurement: return False
    
    # return True
    
    








