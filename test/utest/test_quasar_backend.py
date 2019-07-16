import quasar
import numpy as np
from util.circuit_generator import random_circuit, simple_circuit
from util.error import L1_error


"""
Test "QuasarSimulatorBackend" Class
"""

def util_build_circuit():
    """
    A utility function for the testing functions below.
    It creates a quasar Circuit object.
    """
    circuit = quasar.Circuit(N=2).H(0).CX(0,1)
    
    return circuit
    
    
def quasar_simulator_backend():
    """
    Validate initialization of the "QuasarSimulatorBackend" class.
    """
    backend = quasar.QuasarSimulatorBackend()
    
    if not backend.has_statevector: return False
    if not backend.has_measurement: return False
    
    return True


def build_native_circuit():
    """
    Validate build_native_circuit() in the "QuasarSimulatorBackend" class.
    """

    backend = quasar.QuasarSimulatorBackend()
    circuit1 = util_build_circuit()
    circuit2 = backend.build_native_circuit(circuit1)

    return circuit1==circuit2
    
    
def build_native_circuit_in_basis():
    """
    Validate build_native_circuit_in_basis() in the "QuasarSimulatorBackend" class.
    """

    backend = quasar.QuasarSimulatorBackend()
    circuit1 = util_build_circuit()
    basis = ['Z','X']
    circuit2 = backend.build_native_circuit_in_basis(circuit1, basis)

    return circuit2==circuit1.H(1)
    
    
def build_quasar_circuit():
    """
    Validate build_quasar_circuit() in the "QuasarSimulatorBackend" class.
    """

    backend = quasar.QuasarSimulatorBackend()
    circuit1 = util_build_circuit()
    circuit2 = backend.build_quasar_circuit(circuit1)

    return circuit1==circuit2    
    
    
def run_statevector():
    """
    Validate run_statevector() in the "QuasarSimulatorBackend" class.
    """
    backend = quasar.QuasarSimulatorBackend()
    circuit = util_build_circuit()
    result = backend.run_statevector(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    
    return L1_error((result, ans))   
    
    
def run_measurement():
    """
    Validate run_measurement() in the "QuasarSimulatorBackend" class.
    """
    backend = quasar.QuasarSimulatorBackend()
    circuit = util_build_circuit()
    result = backend.run_measurement(circuit)
    ans = [np.sqrt(1/2),0,0,np.sqrt(1/2)]
    
    # these two lines normalize the result
    result = [result.get('{0:02b}'.format(k), 0) for k in range(4)]
    result = np.array(result)/np.linalg.norm(result)
    
    return L1_error((result, ans), margin=0.1)
    

    
    
    
# print(run_measurement())





















